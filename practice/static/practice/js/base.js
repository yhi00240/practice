/**
  * function for menu
  * menu contaion mainmenu and submenu
  * if user doesn't click mainmenu, submenu doesn't show
*/
$(document).ready( function() {
  var $submenu = $('.submenu');
  var $mainmenu = $('.mainmenu');

  // At first, submenus doesn't show
  $submenu.hide();

  // $submenu.first().delay(400).slideDown(700);

  // if you click a mainmenu item, make this item's attr to chosen
  $submenu.on('click','li', function() {
    $submenu.siblings().find('li').removeClass('chosen');
	$(this).addClass('chosen');
  });

  // if you click a mainmenu item, make the item's submenu to show
  $mainmenu.on('click', 'li', function() {
    $(this).next('.submenu').slideToggle().siblings('.submenu').slideUp();
  });
});